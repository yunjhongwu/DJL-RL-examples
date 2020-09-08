package main.env;

import main.utils.Snapshot;

public interface Environment {

    public void render();

    public Snapshot reset();

    public void seed(long seed);

    public Snapshot step(int action);

    public String name();

}
