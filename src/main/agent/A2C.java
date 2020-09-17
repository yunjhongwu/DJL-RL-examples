package main.agent;

import java.util.Random;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import main.agent.model.DistributionValueModel;
import main.utils.ActionSampler;
import main.utils.Memory;
import main.utils.datatype.Transition;

public class A2C extends Agent {

    private final Random random = new Random(0);
    private final Memory memory = new Memory(1);

    private final int dim_of_state_space;
    private final int num_of_action;
    private final int hidden_size;
    private final float gamma;
    private final Optimizer optimizer;

    private NDManager manager = NDManager.newBaseManager();
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
                    updateModel(submanager);
                }
            }

            NDArray prob = predictor.predict(new NDList(submanager.create(state))).get(0);
            int action = ActionSampler.sampleMultinomial(prob, random);

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
        if (!isEval()) {
            memory.setRewardAndMask(reward, done);
        }
    }

    @Override
    public void reset() {
        if (manager != null) {
            manager.close();
        }
        manager = NDManager.newBaseManager();
        model = DistributionValueModel.newModel(manager, dim_of_state_space, hidden_size, num_of_action);
        predictor = model.newPredictor(new NoopTranslator());

    }

    private void updateModel(NDManager submanager) throws TranslateException {
        Transition transition = memory.get(0);
        NDList net_output = predictor.predict(new NDList(submanager.create(transition.getState())));

        NDArray distribution = net_output.get(0);
        NDArray advantage = net_output.get(1).neg().add(transition.getReward());

        if (!transition.isMasked()) {
            NDArray value_next = predictor.predict(new NDList(submanager.create(transition.getNextState()))).get(1);
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
