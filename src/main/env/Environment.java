package main.env;

import java.util.Random;

import main.utils.Snapshot;

public abstract class Environment {
    protected final Random random = new Random();

    public void seed(long seed) {
        random.setSeed(seed);
    }

    public abstract void render();

    public abstract Snapshot reset();

    public abstract Snapshot step(int action);

    public abstract int DimOfStateSpace();

    public abstract int NumOfActions();

    public abstract double[] getStateSpace(int dim);

}
