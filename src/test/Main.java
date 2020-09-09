package test;

import main.agent.A2C;
import main.env.Environment;
import main.env.cartpole.CartPole;
import main.utils.Runner;

public class Main {
    public static void main(String[] args) {
        Environment env = new CartPole(true);
        Runner runner = new Runner(new A2C(env.DimOfStateSpace(), env.NumOfActions(), 64, 0.99f, 0.001f), env, true);
        runner.run(500);
    }
}
