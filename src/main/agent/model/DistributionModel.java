package main.agent.model;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class DistributionModel extends ScoreModel {
    private DistributionModel(NDManager manager, int hidden_size, int output_size) {
        super(manager, hidden_size, output_size);
    }

    public static Model newModel(NDManager manager, int input_size, int hidden_size, int output_size) {
        Model model = Model.newInstance("DistributionModel");
        AgentModelBlock net = new DistributionModel(manager, hidden_size, output_size);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(input_size));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forward(ParameterStore parameter_store, NDList inputs, boolean training,
            PairList<String, Object> params) {

        return new NDList(super.forward(parameter_store, inputs, training, params).singletonOrThrow().softmax(1));
    }

}
