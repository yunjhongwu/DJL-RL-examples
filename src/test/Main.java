package test;

import main.env.mountaincar.MountainCarVisualizer;

public class Main {
    public static void main(String[] args) {
        // Runner runner = new Runner(new A2C(64, 0.99f, 0.001f), true);
        // runner.run(500);

        var car = new MountainCarVisualizer(-1.2f, 0.6f, 0.5f, 50);

        for (int i = 0; i < 200; i++) {
            car.update(new float[] { -1.2f + i * 0.01f, 0, 0, 0 });
        }
    }
}
