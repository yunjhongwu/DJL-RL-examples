package main.agent;

public abstract class BaseAgent {
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

}
