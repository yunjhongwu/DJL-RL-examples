package main.env.cartpole;

import main.env.Environment;
import main.utils.datatype.Snapshot;

/** Exactly the same environment as CartPole-v1 implemented in gym. */
public final class CartPole extends Environment {
    private static final double[][] STATE_SPACE = new double[][] { { -4.8, 4.8 },
            { Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY }, { -0.418, 0.418 },
            { Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY } };
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

    private final float[] state = new float[] { 0.0f, 0.0f, 0.0f, 0.0f };
    private final CartPoleVisualizer visualizer;

    public CartPole(boolean visual) {
        super(STATE_SPACE);
        visualizer = visual ? new CartPoleVisualizer(LENGTH, X_THRESHOLD, 1000) : null;
    }

    public void render() {
        if (visualizer != null) {
            visualizer.update(state);

        }
    }

    public Snapshot reset() {
        for (int i = 0; i < 4; i++) {
            state[i] = random.nextFloat() * 0.1f - 0.05f;
        }
        return new Snapshot(state, 1.0f, false);
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

        return new Snapshot(state, 1.0f, done);
    }

    @Override
    public int DimOfStateSpace() {
        return 4;
    }

    @Override
    public int NumOfActions() {
        return 2;
    }

}
