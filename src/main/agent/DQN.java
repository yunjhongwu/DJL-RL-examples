package main.agent;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.L2Loss;
import ai.djl.translate.TranslateException;
import main.agent.base.BaseDQN;
import main.utils.ActionSampler;
import main.utils.Helper;
import main.utils.datatype.MemoryBatch;

public class DQN extends BaseDQN {
    private final L2Loss loss_func = new L2Loss();

    public DQN(int dim_of_state_space, int num_of_actions, int hidden_size, int batch_size, int sync_net_interval,
            float gamma, float learning_rate) {
        super(dim_of_state_space, num_of_actions, hidden_size, batch_size, sync_net_interval, gamma, learning_rate);
    }

    @Override
    protected int getAction(NDManager manager, float[] state) throws TranslateException {
        NDArray score = target_predictor.predict(new NDList(manager.create(state))).singletonOrThrow();
        return ActionSampler.epsilonGreedy(score, random, Math.max(MIN_EXPLORE_RATE, epsilon));
    }

    @Override
    protected void updateModel(NDManager manager) throws TranslateException {
        MemoryBatch batch = memory.sampleBatch(batch_size, manager);

        NDArray policy = policy_predictor.predict(new NDList(batch.getStates())).singletonOrThrow();
        NDArray target = target_predictor.predict(new NDList(batch.getNextStates())).singletonOrThrow().duplicate();
        NDArray expected_returns = Helper.gather(policy, batch.getActions().toIntArray());
        NDArray next_returns = batch.getRewards()
                .add(target.max(new int[] { 1 }).mul(batch.getMasks().logicalNot()).mul(gamma));

        NDArray loss = loss_func.evaluate(new NDList(expected_returns), new NDList(next_returns));

        gradientUpdate(loss);

    }

}
