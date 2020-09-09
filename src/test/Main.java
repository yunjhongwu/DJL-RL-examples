package test;

import main.agent.A2C;
import main.env.Environment;
import main.env.mountaincar.MountainCar;
import main.utils.Runner;

public class Main {
    public static void main(String[] args) {
        Environment env = new MountainCar(true);
        Runner runner = new Runner(new A2C(env.DimOfStateSpace(), env.NumOfActions(), 256, 0.99f, 0.00001f), env, true);
        runner.run(500);

    }
}
