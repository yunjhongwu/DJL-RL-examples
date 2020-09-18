package main.agent.model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterType;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

public class DistributionValueModel extends BaseModel {
    private static final float LAYERNORM_MOMENTUM = 0.9999f;
    private static final float LAYERNORM_EPSILON = 1e-5f;
    private final Block linear_input;
    private final Block linear_action;
    private final Block linear_value;

    private final int hidden_size;
    private final int output_size;
    private final Parameter gamma;
    private final Parameter beta;
    private float moving_mean = 0.0f;
    private float moving_var = 1.0f;

    private DistributionValueModel(NDManager manager, int hidden_size, int output_size) {
        super(manager);
        this.linear_input = addChildBlock("linear_input", Linear.builder().setUnits(hidden_size).build());
        this.linear_action = addChildBlock("linear_action", Linear.builder().setUnits(output_size).build());
        this.linear_value = addChildBlock("linear_value", Linear.builder().setUnits(1).build());

        this.gamma = addParameter(new Parameter("mu", this, ParameterType.GAMMA, true), new Shape(1));
        this.beta = addParameter(new Parameter("sigma", this, ParameterType.BETA, true), new Shape(1));

        this.hidden_size = hidden_size;
        this.output_size = output_size;
    }

    public static Model newModel(NDManager manager, int input_size, int hidden_size, int output_size) {
        Model model = Model.newInstance("DistributionValueModel");
        BaseModel net = new DistributionValueModel(manager, hidden_size, output_size);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(input_size));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forward(ParameterStore parameter_store, NDList inputs, boolean training,
            PairList<String, Object> params) {
        NDList hidden = new NDList(
                Activation.relu(linear_input.forward(parameter_store, inputs, training).singletonOrThrow()));

        NDArray distribution = normalize(linear_action.forward(parameter_store, hidden, training).singletonOrThrow())
                .softmax(1);

        NDArray value = linear_value.forward(parameter_store, hidden, training).singletonOrThrow();

        return new NDList(distribution, value);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] input_shape) {
        return new Shape[] { new Shape(output_size), new Shape(1) };
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType data_type, Shape... input_shapes) {
        setInitializer(new XavierInitializer());
        linear_input.initialize(manager, data_type, input_shapes[0]);
        linear_action.initialize(manager, data_type, new Shape(hidden_size));
        linear_value.initialize(manager, data_type, new Shape(hidden_size));
    }

    private NDArray normalize(NDArray arr) {
        float score_mean = arr.mean().getFloat();
        moving_mean = moving_mean * LAYERNORM_MOMENTUM + score_mean * (1.0f - LAYERNORM_MOMENTUM);
        moving_var = moving_var * LAYERNORM_MOMENTUM
                + arr.sub(score_mean).pow(2).mean().getFloat() * (1.0f - LAYERNORM_MOMENTUM);
        return arr.sub(moving_mean).div(Math.sqrt(moving_var + LAYERNORM_EPSILON)).mul(gamma.getArray())
                .add(beta.getArray());
    }

}
