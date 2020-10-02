package main.agent.base;

import java.util.Random;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import main.agent.model.DistributionValueModel;
import main.utils.ActionSampler;
import main.utils.Memory;

public abstract class BaseGAE extends BaseAgent {
    protected final Random random = new Random(0);
    protected final Memory memory = new Memory(1024);

    protected NDManager manager = NDManager.newBaseManager();
    protected Model model;
    protected Predictor<NDList, NDList> predictor;

    private final float gae_lambda;
    private final float gamma;
    private final int num_of_action;
    private final int dim_of_state_space;
    private final int hidden_size;

    public BaseGAE(int dim_of_state_space, int num_of_action, int hidden_size, float gamma, float gae_lambda) {
        this.gae_lambda = gae_lambda;
        this.gamma = gamma;
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_action = num_of_action;
        this.hidden_size = hidden_size;
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
                    // TODO Auto-generated catch block
                    e.printStackTrace();
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

    protected NDList estimateAdvantage(NDArray values, NDArray rewards) {
        NDArray expected_returns = rewards.duplicate();
        NDArray advantages = rewards.sub(values).duplicate();

        for (long i = expected_returns.getShape().get(0) - 2; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            expected_returns.set(index, expected_returns.get(i).add(expected_returns.get(i + 1).mul(gamma)));
            advantages.set(index,
                    advantages.get(i).add(values.get(i + 1).add(advantages.get(i + 1).mul(gae_lambda)).mul(gamma)));
        }

        return new NDList();
    }

    protected abstract void updateModel(NDManager submanager) throws TranslateException;

}
