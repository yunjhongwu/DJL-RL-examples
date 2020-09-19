package test;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import main.agent.A2C;
import main.agent.DQN;
import main.agent.DynaQ;
import main.agent.QRDQN;
import main.agent.model.BaseModel;
import main.env.Environment;
import main.env.cartpole.CartPole;
import main.utils.Runner;

public class Main {
    public static void main(String[] args) throws TranslateException {
        Engine.getInstance().setRandomSeed(0);
//        NDManager manager = NDManager.newBaseManager();
//        Optimizer optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build();
//
//        Model model = TestModel.newModel(manager, 4, 64, 2);
//        Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
//        NDArray X = manager.randomNormal(new Shape(128, 4));
//        NDArray y = manager.randomNormal(new Shape(128, 2));
//        for (int i = 0; i < 55250; i++) {
//            NDArray loss = predictor.predict(new NDList(X)).singletonOrThrow().sub(y).abs().mean();
//            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
//                collector.backward(loss);
//                for (Pair<String, Parameter> params : model.getBlock().getParameters()) {
//                    NDArray params_arr = params.getValue().getArray();
//
//                    optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
//                }
//                System.out.println(i + " " + loss.getFloat());
//
//            }
//        }

        Environment env = new CartPole(false);
        env.seed(0);
        runDQN(env, 500);
    }

    public static void runDynaQ(Environment env, int goal) {
        new Runner(new DynaQ(env.getStateSpace(), env.NumOfActions(), 8, 0.1f, 0.95f, 0.05f, 8), env).run(goal);
    }

    public static void runDQN(Environment env, int goal) {
        new Runner(new DQN(env.DimOfStateSpace(), env.NumOfActions(), 64, 32, 1, 0.95f, 0.001f), env).run(goal);
    }

    public static void runA2C(Environment env, int goal) {
        new Runner(new A2C(env.DimOfStateSpace(), env.NumOfActions(), 64, 0.95f, 0.001f), env).run(goal);
    }

    public static void runQRDQN(Environment env, int goal) {
        new Runner(new QRDQN(env.DimOfStateSpace(), env.NumOfActions(), 8, 64, 32, 32, 0.95f, 0.00001f), env).run(goal);
    }
}

class TestModel extends BaseModel {
    private final Block linear_input;
    private final Block linear_output;

    private final int hidden_size;
    private final int output_size;

    protected TestModel(NDManager manager, int hidden_size, int output_size) {
        super(manager);
        this.linear_input = addChildBlock("linear_input", Linear.builder().setUnits(hidden_size).build());
        this.linear_output = addChildBlock("linear_output", Linear.builder().setUnits(output_size).build());

        this.hidden_size = hidden_size;
        this.output_size = output_size;
    }

    public static Model newModel(NDManager manager, int input_size, int hidden_size, int output_size) {
        Model model = Model.newInstance("ScoreModel");
        BaseModel net = new TestModel(manager, hidden_size, output_size);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(input_size));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forward(ParameterStore parameter_store, NDList inputs, boolean training,
            PairList<String, Object> params) {
        NDList hidden = new NDList(
                Activation.relu(linear_input.forward(parameter_store, inputs, training).singletonOrThrow()));

        return linear_output.forward(parameter_store, hidden, training);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] input_shape) {
        return new Shape[] { new Shape(output_size) };
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType data_type, Shape... input_shapes) {
        setInitializer(new XavierInitializer());
        linear_input.initialize(manager, data_type, input_shapes[0]);
        linear_output.initialize(manager, data_type, new Shape(hidden_size));
    }

}
