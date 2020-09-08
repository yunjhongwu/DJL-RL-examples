package main.env.cartpole;

import java.util.Random;

import main.env.Environment;
import main.utils.Snapshot;

/** Exactly the same environment as CartPole-v1 implemented in gym. */
public final class CartPole implements Environment {
    private static final String NAME = "CartPole-v0";
    private static final double GRAVITY = 9.8;
    private static final double CART_MASS = 1.0;
    private static final double POLE_MASS = 0.1;
    private static final double TOTAL_MASS = CART_MASS + POLE_MASS;
    private static final double LENGTH = 0.5;
    private static final double POLEMASS_LENGTH = POLE_MASS * LENGTH;
    private static final double FORCE_MAG = 10.0;
    private static final double TAU = 0.02;
    private static final double X_THRESHOLD = 2.4;
    private static final double THETA_THRESHOLD = 12 * 2 * Math.PI / 360;

    private final Random random = new Random();
    private final float[] state = new float[] { 0.0F, 0.0F, 0.0F, 0.0F };
    private final CartPoleVisualizer visualizer;

    private int step_beyond_done = -1;

    public CartPole(boolean visual) {
        visualizer = visual ? new CartPoleVisualizer(LENGTH, X_THRESHOLD, 1000) : null;
    }

    public String name() {
        return NAME;
    }

    public void seed(long seed) {
        random.setSeed(seed);
    }

    public void render() {
        if (visualizer != null) {
            visualizer.update(state[0], state[2]);

        }
    }

    public Snapshot reset() {
        for (int i = 0; i < 4; i++) {
            state[i] = random.nextFloat() * 0.1F - 0.05F;
        }
        step_beyond_done = -1;
        return new Snapshot(state, 0.0F, false);
    }

    public Snapshot step(int action) {
        double force = action == 1 ? FORCE_MAG : -FORCE_MAG;
        double cos_theta = Math.cos(state[2]);
        double sin_theta = Math.sin(state[2]);
        double temp = (force + POLEMASS_LENGTH * Math.pow(state[3], 2) * sin_theta) / TOTAL_MASS;

        double theta_acc = ((GRAVITY * sin_theta - temp * cos_theta)
                / (LENGTH * (4.0 / 3.0 - POLE_MASS * Math.pow(cos_theta, 2) / TOTAL_MASS)));
        double x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        state[0] += TAU * state[1];
        state[1] += TAU * x_acc;
        state[2] += TAU * state[3];
        if (state[2] > Math.PI) {
            state[2] -= 2 * Math.PI;
        } else if (state[2] < -Math.PI) {
            state[2] += 2 * Math.PI;
        }
        state[3] += TAU * theta_acc;
        boolean done = (state[0] < -X_THRESHOLD || state[0] > X_THRESHOLD || state[2] < -THETA_THRESHOLD
                || state[2] > THETA_THRESHOLD);
        if (done) {
            step_beyond_done++;
        }

        return new Snapshot(state, step_beyond_done == 0 ? 0 : 1, done);
    }

}
