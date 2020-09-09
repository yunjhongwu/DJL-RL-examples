package main.agent;

import java.util.Random;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.nn.core.Linear;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import main.utils.Memory;
import main.utils.Transition;

public class A2C extends Agent {
    private final float MIN_EXPLORATION = 0.05f;
    private final float EXPLORATION_DECAY = 0.999f;
    private final NDManager manager = NDManager.newBaseManager();
    private final Random random = new Random(0);
    private final Memory memory = new Memory(1);

    private final int dim_of_state_space;
    private final int num_of_action;
    private final int hidden_size;
    private final float gamma;
    private final Optimizer optimizer;

    private float exploration = 1.0f;
    private Model model;
    private Predictor<NDList, NDList> predictor;

    public A2C(int dim_of_state_space, int num_of_action, int hidden_size, float gamma, float learning_rate) {
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_action = num_of_action;
        this.hidden_size = hidden_size;
        this.gamma = gamma;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learning_rate)).build();

        reset();
    }

    @Override
    public int react(float[] state) {
        try (NDManager submanager = manager.newSubManager()) {
            if (!isEval()) {
                memory.setState(state);

                if (memory.size() > 0) {
                    Transition transition = memory.get(0);
                    NDList net_output = predictor.predict(new NDList(submanager.create(transition.getState())));

                    NDArray distribution = net_output.get(0);
                    NDArray advantage = net_output.get(1).neg().add(transition.getReward());

                    if (!transition.isMasked()) {
                        NDArray value_next = predictor.predict(new NDList(submanager.create(transition.getNextState())))
                                .get(1);
                        advantage = advantage.add(value_next.mul(gamma));
                    }

                    NDArray log_distribution = distribution.log();

                    NDArray loss_critic = advantage.square();
                    NDArray loss_actor = log_distribution.get(transition.getAction()).mul(advantage).neg();
                    NDArray loss = loss_actor.add(loss_critic);

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(loss);

                        for (Pair<String, Parameter> params : model.getBlock().getParameters()) {
                            NDArray params_arr = params.getValue().getArray();

                            optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
                        }

                    }
                }
            }

            NDArray prob = predictor.predict(new NDList(submanager.create(state))).get(0);
            int action = 0;
            if (random.nextFloat() < exploration) {
                action = random.nextInt(num_of_action);
            } else {
                float rnd = random.nextFloat();
                while (rnd > 0) {
                    float cut = prob.getFloat(action);
                    if (rnd > cut) {
                        action++;
                    }
                    rnd -= cut;
                }
            }
            if (!isEval()) {
                memory.setAction(action);
            }

            return action;

        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void collect(float reward, boolean done) {
        memory.setRewardAndMask(reward, done);
        if (done) {
            exploration *= EXPLORATION_DECAY;
            exploration = Math.max(MIN_EXPLORATION, exploration);
        }
    }

    @Override
    public void reset() {
        model = PolicyNet.newModel(dim_of_state_space, hidden_size, num_of_action);
        predictor = model.newPredictor(new NoopTranslator());

    }

}

class PolicyNet extends AbstractBlock {
    private static final byte VERSION = 2;
    private static final float LAYERNORM_MOMENTUM = 0.9999f;
    private static final float LAYERNORM_EPSILON = 1e-5f;
    private final Block linear_input;
    private final Block linear_action;
    private final Block linear_value;

    private final int hidden_size;
    private final int output_size;
    private final Parameter gamma;
    private final Parameter beta;
    private NDManager manager = NDManager.newBaseManager();
    private float moving_mean = 0.0f;
    private float moving_var = 1.0f;

    private PolicyNet(int hidden_size, int output_size) {
        super(VERSION);
        linear_input = addChildBlock("linear_input", Linear.builder().setUnits(hidden_size).build());
        linear_action = addChildBlock("linear_action", Linear.builder().setUnits(output_size).build());
        linear_value = addChildBlock("linear_value", Linear.builder().setUnits(1).build());
        gamma = addParameter(new Parameter("mu", this, ParameterType.GAMMA, true), new Shape(1));
        beta = addParameter(new Parameter("sigma", this, ParameterType.BETA, true), new Shape(1));
        this.hidden_size = hidden_size;
        this.output_size = output_size;
    }

    public static Model newModel(int input_size, int hidden_size, int output_size) {
        Model model = Model.newInstance("A2C");
        PolicyNet net = new PolicyNet(hidden_size, output_size);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(input_size));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forward(ParameterStore parameter_store, NDList inputs, boolean training,
            PairList<String, Object> params) {
        NDList hidden = new NDList(
                Activation.relu(linear_input.forward(parameter_store, inputs, training).singletonOrThrow()));

        NDArray score = linear_action.forward(parameter_store, hidden, training).singletonOrThrow();
        float score_mean = score.mean().getFloat();
        moving_mean = moving_mean * LAYERNORM_MOMENTUM + score_mean * (1.0f - LAYERNORM_MOMENTUM);
        moving_var = moving_var * LAYERNORM_MOMENTUM
                + score.sub(score_mean).pow(2).mean().getFloat() * (1.0f - LAYERNORM_MOMENTUM);
        NDArray distribution = score.sub(moving_mean).div(Math.sqrt(moving_var + LAYERNORM_EPSILON))
                .mul(gamma.getArray()).add(beta.getArray()).softmax(1);
        NDArray value = linear_value.forward(parameter_store, hidden, training).singletonOrThrow();

        return new NDList(distribution, value);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] input_shape) {
        return new Shape[] { new Shape(output_size), new Shape(1) };
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType data_type, Shape... input_shapes) {
        setInitializer(new XavierInitializer());
        linear_input.initialize(manager, data_type, input_shapes[0]);
        linear_action.initialize(manager, data_type, new Shape(hidden_size));
        linear_value.initialize(manager, data_type, new Shape(hidden_size));
    }

    private NDManager getManager() {
        return manager;
    }

}
