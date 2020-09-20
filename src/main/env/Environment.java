package main.env;

import java.util.Map;
import java.util.Random;

import main.utils.datatype.Snapshot;

public abstract class Environment {
    protected final Random random = new Random(0);

    private final double[][] state_space;
    private final int dim_of_state_space;
    private final int num_of_actions;

    public Environment(double[][] state_space, int dim_of_state_space, int num_of_actions) {
        if (state_space != null && state_space.length != dim_of_state_space) {
            throw new IllegalArgumentException("Invalid state space and dimension");
        }
        this.state_space = state_space;
        this.dim_of_state_space = dim_of_state_space;
        this.num_of_actions = num_of_actions;
    }

    public Environment(Map<String, Object> attributes) {
        this((double[][]) attributes.get("state_space"), (int) attributes.get("dim_of_state_space"),
                (int) attributes.get("num_of_actions"));
    }

    /**
     * Reset the seed.
     * 
     * @param seed
     */
    public void seed(long seed) {
        random.setSeed(seed);
    }

    /**
     * Get the range of the given dimension.
     * 
     * @param dimension
     * @return range, represented by a double array of length 2
     */
    public final double[] getStateSpace(int dim) {
        if (state_space == null) {
            throw new UnsupportedOperationException("State space has not been specified.");
        }
        if (dim < 0 || dim >= state_space.length) {
            throw new IllegalArgumentException("Dimension is between 0 and " + DimOfStateSpace());
        }
        return state_space[dim].clone();
    }

    /**
     * Get the range of all the coordinates.
     * 
     * @return ranges, represented by a n by 2 double array
     */
    public final double[][] getStateSpace() {
        if (state_space == null) {
            throw new UnsupportedOperationException("State space has not been specified.");
        }

        double[][] space = new double[state_space.length][];
        for (int i = 0; i < state_space.length; i++) {
            space[i] = state_space[i].clone();
        }
        return space;
    }

    /**
     * Return the dimension of the state space.
     * 
     * @return dimension
     */
    public final int DimOfStateSpace() {
        return dim_of_state_space;
    }

    /**
     * Return the number of the possible actions.
     * 
     * @return number of actions
     */
    public final int NumOfActions() {
        return num_of_actions;
    }

    public abstract void render();

    /**
     * Reset the environment with a random state.
     * 
     * @return
     */
    public abstract Snapshot reset();

    /**
     * Change the internal state according to the input action and return a
     * snapshot.
     * 
     * @param action
     * @return snapshot
     */
    public abstract Snapshot step(int action);

}
