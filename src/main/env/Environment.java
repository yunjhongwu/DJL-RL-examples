package main.env;

import java.util.Random;

import main.utils.datatype.Snapshot;

public abstract class Environment {
    protected final Random random = new Random(0);
    protected final double[][] state_space;

    public Environment(double[][] state_space) {
        this.state_space = state_space;
    }

    public void seed(long seed) {
        random.setSeed(seed);
    }

    public double[] getStateSpace(int dim) {
        if (dim < 0 || dim >= state_space.length) {
            throw new IllegalArgumentException("Dimension is between 0 and " + DimOfStateSpace());
        }
        return state_space[dim].clone();
    }

    public double[][] getStateSpace() {
        double[][] space = new double[state_space.length][];
        for (int i = 0; i < state_space.length; i++) {
            space[i] = state_space[i].clone();
        }
        return space;
    }

    public abstract void render();

    public abstract Snapshot reset();

    public abstract Snapshot step(int action);

    public abstract int DimOfStateSpace();

    public abstract int NumOfActions();

}
