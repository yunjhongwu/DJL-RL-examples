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
import main.agent.base.BaseGAE;
import main.agent.model.DistributionValueModel;
import main.utils.ActionSampler;
import main.utils.Helper;
import main.utils.Memory;
import main.utils.datatype.MemoryBatch;

public class GAE extends BaseGAE {

    private final Random random = new Random(0);
    private final Memory memory = new Memory(1024);

    private final int dim_of_state_space;
    private final int num_of_action;
    private final int hidden_size;
    private final Optimizer optimizer;

    private NDManager manager = NDManager.newBaseManager();
    private Model model;
    private Predictor<NDList, NDList> predictor;

    public GAE(int dim_of_state_space, int num_of_action, int hidden_size, float gamma, float gae_lambda,
            float learning_rate) {
        super(gamma, gae_lambda);
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_action = num_of_action;
        this.hidden_size = hidden_size;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learning_rate)).build();

        reset();
    }

    @Override
    public int react(float[] state) {
        try (NDManager submanager = manager.newSubManager()) {
            if (!isEval()) {
                memory.setState(state);
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
            if (done) {
                try (NDManager submanager = manager.newSubManager()) {
                    updateModel(submanager);
                } catch (TranslateException e) {
                    throw new IllegalStateException(e);
                }
                memory.reset();
            }
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
        MemoryBatch transition = memory.getOrderedBatch(submanager);
        NDList net_output = predictor.predict(new NDList(transition.getStates()));

        NDArray distribution = net_output.get(0);
        NDArray values = net_output.get(1);
        NDArray rewards = transition.getRewards();
        NDList estimates = estimateAdvantage(values, rewards);
        NDArray expected_returns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        NDArray log_distribution = distribution.log();
        NDArray loss_critic = (expected_returns.sub(values)).square().sum();
        NDArray loss_actor = Helper.gather(log_distribution, transition.getActions().toIntArray()).mul(advantages).neg()
                .sum();
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
