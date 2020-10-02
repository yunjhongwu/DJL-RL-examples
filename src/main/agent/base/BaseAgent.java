package main.agent.base;

public abstract class BaseAgent {
    private boolean is_eval = false;

    /**
     * Calculate the action to the input state
     * 
     * @param state
     * @return action
     */
    public abstract int react(float[] state);

    /**
     * Collect the result of the previous action
     * 
     * @param reward
     * @param done
     */
    public abstract void collect(float reward, boolean done);

    /**
     * Reset the agent.
     */
    public abstract void reset();

    /**
     * Switch to the training mode.
     */
    public final void train() {
        this.is_eval = false;
    }

    /**
     * Switch to the inference mode.
     */
    public final void eval() {
        this.is_eval = true;
    }

    /**
     * Check if the agent is in the inference mode.
     * 
     * @return true if the agent is in the inference mode.
     */
    public final boolean isEval() {
        return is_eval;
    }

}
