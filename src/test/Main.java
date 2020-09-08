package test;

import main.agent.A2C;
import main.agent.Agent;
import main.env.Environment;
import main.env.cartpole.CartPole;
import main.utils.Snapshot;

public class Main {
    public static void main(String[] args) {
        Environment env = new CartPole(true);
        Agent agent = new A2C(64, 0.99f, 0.001f);
        double score = 0.0;
        int epoch = 0;

        env.seed(0);

        while (score < 500) {
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
