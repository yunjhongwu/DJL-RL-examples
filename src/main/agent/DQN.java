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
import ai.djl.training.loss.L2Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import main.agent.model.ScoreModel;
import main.utils.ActionSampler;
import main.utils.Helper;
import main.utils.Memory;
import main.utils.datatype.Batch;

public class DQN extends Agent {
    protected final float MIN_EXPLORE_RATE = 0.05f;
    protected final float DECAY_EXPLORE_RATE = 0.999f;
    protected final Random random = new Random(0);
    protected final Memory memory = new Memory(4096, true);
    protected final L2Loss loss_func = new L2Loss();

    private final int dim_of_state_space;
    private final int num_of_actions;
    private final int hidden_size;
    protected final int batch_size;
    protected final int sync_net_interval;
    protected final float gamma;
    protected final Optimizer optimizer;

    protected NDManager manager;
    protected Model policy_net;
    protected Model target_net;
    protected Predictor<NDList, NDList> policy_predictor;
    protected Predictor<NDList, NDList> target_predictor;
    protected int iteration = 0;
    protected float epsilon = 1.0f;

    public DQN(int dim_of_state_space, int num_of_actions, int hidden_size, int batch_size, int sync_net_interval,
            float gamma, float learning_rate) {
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_actions = num_of_actions;
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

                if (memory.size() > batch_size) {
                    updateModel(submanager);
                }
            }

            NDArray score = policy_predictor.predict(new NDList(submanager.create(state))).singletonOrThrow();
            int action = ActionSampler.epsilonGreedy(score, random, Math.max(MIN_EXPLORE_RATE, epsilon));

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
        policy_net = ScoreModel.newModel(manager, dim_of_state_space, hidden_size, num_of_actions);
        target_net = ScoreModel.newModel(manager, dim_of_state_space, hidden_size, num_of_actions);
        policy_predictor = policy_net.newPredictor(new NoopTranslator());
        target_predictor = target_net.newPredictor(new NoopTranslator());

        syncNets();
    }

    protected void syncNets() {
        for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
            target_net.getBlock().getParameters().get(params.getKey())
                    .setArray(params.getValue().getArray().duplicate());

        }
        target_predictor = target_net.newPredictor(new NoopTranslator());
    }

    private void updateModel(NDManager submanager) throws TranslateException {
        Batch batch = memory.sampleBatch(batch_size, submanager);
        NDArray policy = policy_predictor.predict(new NDList(batch.getStates())).singletonOrThrow();
        NDArray target = target_predictor.predict(new NDList(batch.getNextStates())).singletonOrThrow();
        NDArray expected_returns = Helper.gather(policy, batch.getActions().toIntArray());
        NDArray next_returns = batch.getRewards()
                .add(target.max(new int[] { 1 }).mul(batch.getMasks().logicalNot()).mul(gamma)).duplicate();
        NDArray loss = loss_func.evaluate(new NDList(expected_returns), new NDList(next_returns)).mul(2);

        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            collector.backward(loss);
            for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
                NDArray params_arr = params.getValue().getArray();
                optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
            }

        }

        if (iteration++ % sync_net_interval == 0) {
            epsilon *= DECAY_EXPLORE_RATE;
            syncNets();
        }
    }

}
