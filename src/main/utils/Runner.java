package main.utils;

import main.agent.Agent;
import main.env.Environment;
import main.env.cartpole.CartPole;

public class Runner {
    private final Agent agent;
    private final Environment env;

    public Runner(Agent agent, boolean visual) {
        this.agent = agent;
        this.env = new CartPole(visual);
    }

    public void run(double goal) {
        double score = 0.0;
        int epoch = 0;

        env.seed(0);

        while (score < goal) {
            epoch++;
            Snapshot snapshot = env.reset();
            boolean done = false;
            int episode_score = 0;

            while (!done) {
                episode_score++;
                env.render();
                snapshot = env.step(agent.react(snapshot.getState()));
                done = snapshot.isMasked();
                agent.collect(snapshot.getReward(), done);

            }

            score = score * 0.95 + episode_score * 0.05;
            System.out.printf("Epoch %d (%d): %.2f\n", epoch, episode_score, score);
        }

    }
}
