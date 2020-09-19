package main.utils.datatype;

import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

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
        try {
            return new ObjectMapper().writeValueAsString(Map.of("state", state, "reward", reward, "mask", mask));
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
