package main.agent;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import main.agent.base.BaseGAE;
import main.utils.Helper;
import main.utils.datatype.MemoryBatch;

public class PPO extends BaseGAE {
    private final int inner_updates;
    private final int inner_batch_size;
    private final float ratio_lower_bound;
    private final float ratio_upper_bound;

    public PPO(int dim_of_state_space, int num_of_action, int hidden_size, float gamma, float gae_lambda,
            float learning_rate, int inner_updates, int inner_batch_size, float ratio_clip) {
        super(dim_of_state_space, num_of_action, hidden_size, gamma, gae_lambda, learning_rate);
        this.inner_updates = inner_updates;
        this.inner_batch_size = inner_batch_size;
        this.ratio_lower_bound = 1.0f - ratio_clip;
        this.ratio_upper_bound = 1.0f + ratio_clip;
    }

    @Override
    protected void updateModel(NDManager submanager) throws TranslateException {
        MemoryBatch batch = memory.getOrderedBatch(submanager);
        NDArray states = batch.getStates();
        NDArray actions = batch.getActions();

        NDList net_output = predictor.predict(new NDList(states));

        NDArray distribution = Helper.gather(net_output.get(0).duplicate(), actions.toIntArray());
        NDArray values = net_output.get(1).duplicate();

        NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards());
        NDArray expected_returns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        int[] index = new int[inner_batch_size];

        for (int i = 0; i < inner_updates * (1 + batch.size() / inner_batch_size); i++) {
            for (int j = 0; j < inner_batch_size; j++) {
                index[j] = random.nextInt(batch.size());
            }
            NDArray states_subset = getSample(submanager, states, index);
            NDArray actions_subset = getSample(submanager, actions, index);
            NDArray distribution_subset = getSample(submanager, distribution, index);
            NDArray expected_returns_subset = getSample(submanager, expected_returns, index);
            NDArray advantages_subset = getSample(submanager, advantages, index);

            NDList net_output_updated = predictor.predict(new NDList(states_subset));
            NDArray distribution_updated = Helper.gather(net_output_updated.get(0), actions_subset.toIntArray());
            NDArray values_updated = net_output_updated.get(1);

            NDArray loss_critic = (expected_returns_subset.sub(values_updated)).square().sum();

            NDArray ratios = distribution_updated.div(distribution_subset).expandDims(1);

            NDArray loss_actor = ratios.clip(ratio_lower_bound, ratio_upper_bound).mul(advantages_subset)
                    .minimum(ratios.mul(advantages_subset)).sum().neg();
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

    private NDArray getSample(NDManager submanager, NDArray array, int[] index) {

        Shape shape = Shape.update(array.getShape(), 0, inner_batch_size);
        NDArray sample = submanager.zeros(shape, array.getDataType());
        for (int i = 0; i < inner_batch_size; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }

}
