package main.utils;

import main.agent.Agent;
import main.env.Environment;

public class Runner {
    private final Agent agent;
    private final Environment env;

    public Runner(Agent agent, Environment env) {
        this.agent = agent;
        this.env = env;
    }

    public void run(double goal) {
        double score = Double.NEGATIVE_INFINITY;
        int epoch = 0;

        env.seed(0);

        while (score < goal) {
            epoch++;
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

            System.out.printf("Epoch %d (%d): %.2f\n", epoch, episode_score, score);

        }

    }
}
