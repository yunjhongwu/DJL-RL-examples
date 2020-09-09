package main.utils;

import java.util.Arrays;

public final class Transition extends Snapshot {

    private final float[] state_next;
    private final int action;

    public Transition(float[] state, float[] state_next, int action, float reward, boolean mask) {
        super(state, reward, mask);
        this.state_next = state_next != null ? state_next.clone() : null;
        this.action = action;
    }

    public float[] getNextState() {
        return state_next;
    }

    public int getAction() {
        return action;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder(super.toString());
        builder.deleteCharAt(builder.length() - 1);
        builder.append(",\"state_next\":");
        builder.append(Arrays.toString(state_next));
        builder.append(",\"action\":");
        builder.append(action);
        builder.append('}');

        return builder.toString();
    }

}
