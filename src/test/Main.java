package test;

import main.agent.A2C;
import main.utils.Runner;

public class Main {
    public static void main(String[] args) {
        Runner runner = new Runner(new A2C(64, 0.99f, 0.001f), true);
        runner.run(500);
    }
}
