package main.env.mountaincar;

import main.env.Environment;
import main.utils.datatype.Snapshot;

/** Exactly the same environment as MountainCar-v0 implemented in gym. */
public final class MountainCar extends Environment {
    private static final double[][] STATE_SPACE = new double[][] { { -1.2, 0.6 }, { -0.07, 0.07 } };
    private static final float MIN_POSITION = -1.2f;
    private static final float MAX_POSITION = 0.6f;
    private static final float MAX_SPEED = 0.1f;
    private static final float GOAL_POSITION = 0.5f;
    private static final float GOAL_VELOCITY = 0.0f;
    private static final double FORCE = 0.001;
    private static final double GRAVITY = 0.0025;
    private static final double MAX_EPISODE_LENGTH = 200;

    private final float[] state = new float[] { 0.0f, 0.0f };
    private final MountainCarVisualizer visualizer;

    private int episode_length = 0;

    public MountainCar(boolean visual) {
        super(STATE_SPACE);
        visualizer = visual ? new MountainCarVisualizer(MIN_POSITION, MAX_POSITION, GOAL_POSITION, 1000) : null;
    }

    public void render() {
        if (visualizer != null) {
            visualizer.update(state);
        }
    }

    public Snapshot reset() {
        state[0] = random.nextFloat() * 0.2f - 0.6f;
        episode_length = 0;

        return new Snapshot(state, -1.0f, false);
    }

    public Snapshot step(int action) {
        state[1] += (action - 1) * FORCE - Math.cos(3 * state[0]) * GRAVITY;
        state[1] = Math.min(Math.max(state[1], -MAX_SPEED), MAX_SPEED);
        state[0] += state[1];
        state[0] = Math.min(Math.max(state[0], MIN_POSITION), MAX_POSITION);

        if (state[0] < MIN_POSITION) {
            state[0] = MIN_POSITION;
            if (state[1] < 0) {
                state[1] = 0.0f;
            }
        }
        boolean done = ((state[0] >= GOAL_POSITION && state[1] >= GOAL_VELOCITY)
                || ++episode_length >= MAX_EPISODE_LENGTH);

        return new Snapshot(state, -1, done);
    }

    @Override
    public int DimOfStateSpace() {
        return 2;
    }

    @Override
    public int NumOfActions() {
        return 3;
    }

}
