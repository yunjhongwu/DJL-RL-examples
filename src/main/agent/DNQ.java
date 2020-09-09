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
import ai.djl.training.loss.L2Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import main.utils.Memory;
import main.utils.MultinomialSampler;

public class DNQ extends Agent {
    private final NDManager manager = NDManager.newBaseManager();
    private final Random random = new Random(0);
    private final Memory memory = new Memory(4096, true);
    private final L2Loss loss_func = new L2Loss();

    private final int dim_of_state_space;
    private final int num_of_action;
    private final int hidden_size;
    private final int batch_size;
    private final int sync_net_interval;
    private final float gamma;
    private final Optimizer optimizer;

    private Model policy_net;
    private Model target_net;
    private Predictor<NDList, NDList> policy_predictor;
    private Predictor<NDList, NDList> target_predictor;
    private int iteration = 0;

    public DNQ(int dim_of_state_space, int num_of_action, int hidden_size, int batch_size, int sync_net_interval,
            float gamma, float learning_rate) {
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_action = num_of_action;
        this.hidden_size = hidden_size;
        this.batch_size = batch_size;
        this.sync_net_interval = sync_net_interval;
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
                    NDList batch = memory.sampleBatch(batch_size, submanager);
                    NDArray policy = policy_predictor.predict(new NDList(batch.get(0))).singletonOrThrow();
                    NDArray target = target_predictor.predict(new NDList(batch.get(1))).singletonOrThrow();
                    NDArray loss = loss_func.evaluate(new NDList(gather(policy, batch.get(2).toIntArray())),
                            new NDList(target.max(new int[] { 1 }).duplicate().mul(batch.get(4).logicalNot()).mul(gamma)
                                    .add(batch.get(3))));

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(loss);

                        for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
                            NDArray params_arr = params.getValue().getArray();

                            optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
                        }

                    }
                }

                if (++iteration % sync_net_interval == 0) {
                    syncNets();
                }
            }

            NDArray prob = policy_predictor.predict(new NDList(submanager.create(state))).singletonOrThrow();
            int action = MultinomialSampler.sample(prob, random);

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
    }

    @Override
    public void reset() {
        policy_net = DNQPolicyNet.newModel(dim_of_state_space, hidden_size, num_of_action);
        target_net = DNQPolicyNet.newModel(dim_of_state_space, hidden_size, num_of_action);
        policy_predictor = policy_net.newPredictor(new NoopTranslator());
        target_predictor = target_net.newPredictor(new NoopTranslator());
        syncNets();
    }

    private void syncNets() {
        for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
            target_net.getBlock().getParameters().get(params.getKey())
                    .setArray(params.getValue().getArray().duplicate());
        }

    }

    private NDArray gather(NDArray arr, int[] indexes) {
        boolean[][] mask = new boolean[(int) arr.size(0)][(int) arr.size(1)];
        for (int i = 0; i < indexes.length; i++) {
            mask[i][indexes[i]] = true;
        }

        return arr.get(arr.getManager().create(mask));
    }
}

class DNQPolicyNet extends AbstractBlock {
    private static final byte VERSION = 2;
    private static final float LAYERNORM_MOMENTUM = 0.9999f;
    private static final float LAYERNORM_EPSILON = 1e-5f;
    private final Block linear_input;
    private final Block linear_output;

    private final int hidden_size;
    private final int output_size;
    private final Parameter gamma;
    private final Parameter beta;
    private NDManager manager = NDManager.newBaseManager();
    private float moving_mean = 0.0f;
    private float moving_var = 1.0f;

    private DNQPolicyNet(int hidden_size, int output_size) {
        super(VERSION);

        this.linear_input = addChildBlock("linear_input", Linear.builder().setUnits(hidden_size).build());
        this.linear_output = addChildBlock("linear_output", Linear.builder().setUnits(output_size).build());
        this.gamma = addParameter(new Parameter("mu", this, ParameterType.GAMMA, true), new Shape(1));
        this.beta = addParameter(new Parameter("sigma", this, ParameterType.BETA, true), new Shape(1));

        this.hidden_size = hidden_size;
        this.output_size = output_size;
    }

    public static Model newModel(int input_size, int hidden_size, int output_size) {
        Model model = Model.newInstance("A2C");
        DNQPolicyNet net = new DNQPolicyNet(hidden_size, output_size);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(input_size));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forward(ParameterStore parameter_store, NDList inputs, boolean training,
            PairList<String, Object> params) {
        NDList hidden = new NDList(
                Activation.relu(linear_input.forward(parameter_store, inputs, training).singletonOrThrow()));

        NDArray distribution = normalize(linear_output.forward(parameter_store, hidden, training).singletonOrThrow())
                .softmax(1);

        return new NDList(distribution);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] input_shape) {
        return new Shape[] { new Shape(output_size) };
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType data_type, Shape... input_shapes) {
        setInitializer(new XavierInitializer());
        linear_input.initialize(manager, data_type, input_shapes[0]);
        linear_output.initialize(manager, data_type, new Shape(hidden_size));
    }

    private NDManager getManager() {
        return manager;
    }

    private NDArray normalize(NDArray arr) {
        float score_mean = arr.mean().getFloat();
        moving_mean = moving_mean * LAYERNORM_MOMENTUM + score_mean * (1.0f - LAYERNORM_MOMENTUM);
        moving_var = moving_var * LAYERNORM_MOMENTUM
                + arr.sub(score_mean).pow(2).mean().getFloat() * (1.0f - LAYERNORM_MOMENTUM);
        return arr.sub(moving_mean).div(Math.sqrt(moving_var + LAYERNORM_EPSILON)).mul(gamma.getArray())
                .add(beta.getArray());
    }

}
