package main.agent;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import main.agent.base.BaseDQN;
import main.utils.ActionSampler;
import main.utils.Helper;
import main.utils.datatype.MemoryBatch;

public class QRDQN extends BaseDQN {
    private final int num_of_actions;
    private final int num_of_action_bins;
    private final NDArray quantiles;

    public QRDQN(int dim_of_state_space, int num_of_actions, int num_of_action_bins, int hidden_size, int batch_size,
            int sync_net_interval, float gamma, float learning_rate) {
        super(dim_of_state_space, num_of_actions * num_of_action_bins, hidden_size, batch_size, sync_net_interval,
                gamma, learning_rate);
        this.num_of_actions = num_of_actions;
        this.num_of_action_bins = num_of_action_bins;

        float[] quantiles = new float[num_of_action_bins];

        float grid = 0.5f / num_of_action_bins;
        float interval = 1.0f / num_of_action_bins;

        for (int i = 0; i < num_of_action_bins; i++) {
            quantiles[i] = grid;
            grid += interval;
        }
        this.quantiles = manager.create(quantiles);

    }

    @Override
    protected int getAction(NDManager manager, float[] state) throws TranslateException {
        NDArray score = target_predictor.predict(new NDList(manager.create(state))).singletonOrThrow()
                .reshape(num_of_actions, num_of_action_bins).mean(new int[] { 1 });
        return ActionSampler.epsilonGreedy(score, random, Math.max(MIN_EXPLORE_RATE, epsilon));
    }

    int update = 0;

    @Override
    protected void updateModel(NDManager manager) throws TranslateException {
        MemoryBatch batch = memory.sampleBatch(batch_size, manager);

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
        Shape extended_shape = expected_returns.getShape().add(num_of_action_bins);

        NDArray residuals = Helper.tile(next_returns, extended_shape)
                .sub(Helper.tile(expected_returns, extended_shape).swapAxes(1, 2));

        NDArray sq_loss_area = residuals.abs().lt(1);
        NDArray huber = residuals.abs().sub(0.5f).mul(sq_loss_area.logicalNot())
                .add(residuals.pow(2).mul(0.5).mul(sq_loss_area));
        NDArray loss = huber.mul(quantiles.sub(residuals.lt(0).toType(DataType.FLOAT32, false)).abs()).mean();

        gradientUpdate(loss);

    }

}
