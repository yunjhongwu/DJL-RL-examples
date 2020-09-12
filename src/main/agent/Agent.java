package main.agent;

import ai.djl.ndarray.NDArray;

public abstract class Agent {
    private boolean is_eval = false;

    public abstract int react(float[] state);

    public abstract void collect(float reward, boolean done);

    public abstract void reset();

    public final void train() {
        this.is_eval = false;
    }

    public final void eval() {
        this.is_eval = true;
    }

    public final boolean isEval() {
        return is_eval;
    }

    protected static NDArray gather(NDArray arr, int[] indexes) {
        boolean[][] mask = new boolean[(int) arr.size(0)][(int) arr.size(1)];
        for (int i = 0; i < indexes.length; i++) {
            mask[i][indexes[i]] = true;
        }

        return arr.get(arr.getManager().create(mask));
    }
}
