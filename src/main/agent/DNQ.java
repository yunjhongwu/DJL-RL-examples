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
import main.utils.Memory;
import main.utils.MultinomialSampler;

public class DNQ extends Agent {
    private final float MIN_EXPLORE_RATE = 0.05f;
    private final float DECAY_EXPLORE_RATE = 0.99f;
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

    private NDManager manager;
    private Model policy_net;
    private Model target_net;
    private Predictor<NDList, NDList> policy_predictor;
    private Predictor<NDList, NDList> target_predictor;
    private int iteration = 0;
    private float exploration = 1.0f;

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
                    updateModel(submanager);
                }

            }

            NDArray score = policy_predictor.predict(new NDList(submanager.create(state))).singletonOrThrow();
            int action = MultinomialSampler.exploreExploit(score, random, Math.max(MIN_EXPLORE_RATE, exploration));

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
        manager = NDManager.newBaseManager();
        policy_net = ScoreModel.newModel(manager, dim_of_state_space, hidden_size, num_of_action);
        target_net = ScoreModel.newModel(manager, dim_of_state_space, hidden_size, num_of_action);
        policy_predictor = policy_net.newPredictor(new NoopTranslator());
        target_predictor = target_net.newPredictor(new NoopTranslator());
        syncNets();
    }

    private void updateModel(NDManager submanager) throws TranslateException {
        NDList batch = memory.sampleBatch(batch_size, submanager);

        NDArray policy = policy_predictor.predict(new NDList(batch.get(0))).singletonOrThrow();
        NDArray target = target_predictor.predict(new NDList(batch.get(1))).singletonOrThrow();

        NDArray loss = loss_func.evaluate(new NDList(gather(policy, batch.get(2).toIntArray())), new NDList(
                target.max(new int[] { 1 }).duplicate().mul(batch.get(4).logicalNot()).mul(gamma).add(batch.get(3))));
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            collector.backward(loss);

            for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
                NDArray params_arr = params.getValue().getArray();
                optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
            }

        }

        if (++iteration % sync_net_interval == 0) {
            exploration *= DECAY_EXPLORE_RATE;
            syncNets();
        }
    }

    private void syncNets() {
        for (Pair<String, Parameter> params : policy_net.getBlock().getParameters()) {
            target_net.getBlock().getParameters().get(params.getKey())
                    .setArray(params.getValue().getArray().duplicate());
        }

    }

}
