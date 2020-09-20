package main.utils;

import main.agent.BaseAgent;
import main.env.Environment;
import main.utils.datatype.Snapshot;

public final class Runner {
    private final BaseAgent agent;
    private final Environment env;

    public Runner(BaseAgent agent, Environment env) {
        this.agent = agent;
        this.env = env;
    }

    public void run(double goal) {
        double score = Double.NEGATIVE_INFINITY;
        int episode = 0;

        while (score < goal) {
            episode++;
            Snapshot snapshot = env.reset();
            boolean done = false;
            int episode_score = 0;

            while (!done) {
                env.render();
                snapshot = env.step(agent.react(snapshot.getState()));
                done = snapshot.isMasked();
                episode_score += snapshot.getReward();
                agent.collect(snapshot.getReward(), done);

            }
            score = score > Double.NEGATIVE_INFINITY ? score * 0.95 + episode_score * 0.05 : episode_score;
            System.out.printf("Epoch %d (%d): %.2f\n", episode, episode_score, score);

        }

    }
}
