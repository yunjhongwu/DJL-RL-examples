package main.agent;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslateException;
import main.utils.ActionSampler;
import main.utils.Helper;
import main.utils.datatype.Batch;

public class QRDQN extends BaseDQN {
    private final int num_of_actions;
    private final int num_of_action_bins;

    public QRDQN(int dim_of_state_space, int num_of_actions, int num_of_action_bins, int hidden_size, int batch_size,
            int sync_net_interval, float gamma, float learning_rate) {
        super(dim_of_state_space, num_of_actions * num_of_action_bins, hidden_size, batch_size, sync_net_interval,
                gamma, learning_rate);
        this.num_of_actions = num_of_actions;
        this.num_of_action_bins = num_of_action_bins;
    }

    @Override
    protected int getAction(NDManager manager, float[] state) throws TranslateException {
        NDArray score = policy_predictor.predict(new NDList(manager.create(state))).singletonOrThrow()
                .reshape(num_of_actions, num_of_action_bins).mean(new int[] { 1 });
        return ActionSampler.epsilonGreedy(score, random, Math.max(MIN_EXPLORE_RATE, epsilon));
    }

    @Override
    protected void updateModel(NDManager manager) throws TranslateException {
        Batch batch = memory.sampleBatch(batch_size, manager);
        NDArray policy = policy_predictor.predict(new NDList(batch.getStates())).singletonOrThrow().reshape(-1,
                num_of_actions, num_of_action_bins);
        NDArray target = target_predictor.predict(new NDList(batch.getNextStates())).singletonOrThrow().reshape(-1,
                num_of_actions, num_of_action_bins);

        NDArray next_actions = target.mean(new int[] { 2 }).argMax(1).toType(DataType.INT32, false);

        NDArray expected_returns = Helper.gather(policy, batch.getActions().toIntArray());
        NDArray next_returns = Helper.tile(batch.getRewards(), expected_returns.getShape())
                .add(Helper.gather(target, next_actions.toIntArray())
                        .mul(Helper.tile(batch.getMasks().logicalNot(), expected_returns.getShape())).mul(gamma))
                .duplicate();

        NDArray loss = loss_func.evaluate(new NDList(expected_returns), new NDList(next_returns));

        gradientUpdate(loss);

    }

}
