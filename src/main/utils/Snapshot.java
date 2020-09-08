package main.utils;

import java.util.Arrays;

public class Snapshot {
    private final float[] state;
    private final float reward;
    private final boolean mask;

    public Snapshot(float[] state, float reward, boolean mask) {
        this.state = state.clone();
        this.reward = reward;
        this.mask = mask;
    }

    public float[] getState() {
        return state;
    }

    public float getReward() {
        return reward;
    }

    public boolean isMasked() {
        return mask;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("{\"state\":");
        builder.append(Arrays.toString(state));
        builder.append(",\"reward\":");
        builder.append(reward);
        builder.append(",\"mask\":");
        builder.append(mask);
        builder.append('}');

        return builder.toString();
    }
}
