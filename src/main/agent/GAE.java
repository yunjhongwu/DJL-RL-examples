package main.agent;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import main.agent.base.BaseGAE;
import main.utils.Helper;
import main.utils.datatype.MemoryBatch;

public class GAE extends BaseGAE {
    public GAE(int dim_of_state_space, int num_of_action, int hidden_size, float gamma, float gae_lambda,
            float learning_rate) {
        super(dim_of_state_space, num_of_action, hidden_size, gamma, gae_lambda, learning_rate);

        reset();
    }

    protected void updateModel(NDManager submanager) throws TranslateException {
        MemoryBatch batch = memory.getOrderedBatch(submanager);

        NDList net_output = predictor.predict(new NDList(batch.getStates()));
        NDArray distribution = net_output.get(0);
        NDArray values = net_output.get(1);
        NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards());
        NDArray expected_returns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        NDArray loss_critic = (expected_returns.sub(values)).square().sum();
        NDArray loss_actor = Helper.gather(distribution.log(), batch.getActions().toIntArray()).mul(advantages).sum()
                .neg();
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
