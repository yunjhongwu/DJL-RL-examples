package main.agent;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import main.utils.Memory;
import main.utils.Transition;

public class DynaQ extends Agent {
    private final Random random = new Random(0);
    private final Memory memory = new Memory(1);
    private final EmpiricalModel env;
    private final int[] index;
    private final float[][] table;

    private final double[][] state_ranges;

    private final int state_resolution;
    private final int num_of_states;
    private final int num_of_actions;
    private final float alpha;
    private final float gamma;
    private final double epsilon;
    private final int num_of_planning_iterations;

    public DynaQ(double[][] state_ranges, int state_resolution, int num_of_actions, float alpha, float gamma,
            float epsilon, int num_of_planning_iterations) {
        int num_of_states = 1;

        this.state_ranges = new double[state_ranges.length][];
        for (int i = 0; i < state_ranges.length; i++) {
            double[] range = state_ranges[i];
            this.state_ranges[i] = state_ranges[i].clone();

            if (range == null || range.length != 2) {
                throw new IllegalArgumentException("Invalid state range");
            }

            num_of_states *= state_resolution;
        }

        this.num_of_states = num_of_states;

        this.env = new EmpiricalModel(num_of_states, num_of_actions, random);

        this.index = IntStream.range(0, num_of_actions).toArray();
        this.state_resolution = state_resolution;
        this.num_of_actions = num_of_actions;
        this.gamma = gamma;
        this.alpha = alpha;
        this.epsilon = epsilon;
        this.num_of_planning_iterations = num_of_planning_iterations;

        table = new float[num_of_states][num_of_actions];

    }

    @Override
    public int react(float[] state) {
        if (!isEval()) {
            memory.setState(state);
            if (memory.size() > 0) {
                Transition transition = memory.get(0);
                int state_value = encodeState(transition.getState());
                int state_next = encodeState(transition.getNextState());
                env.update(state_value, state_next, transition.getAction(), transition.getReward(),
                        transition.isMasked());
                updateTable(state_value, state_next, transition.getAction(), transition.getReward(),
                        transition.isMasked());
            }
        }

        int action = (random.nextDouble() < epsilon || memory.size() == 0 ? random.nextInt(num_of_actions)
                : getRandomMaxPolicy(encodeState(state)));

        if (!isEval()) {
            memory.setAction(action);
            if (memory.size() > 0) {
                planning();
            }
        }

        return action;
    }

    @Override
    public void collect(float reward, boolean done) {
        if (!isEval()) {
            memory.setRewardAndMask(reward, done);
        }
    }

    @Override
    public void reset() {
        for (int i = 0; i < num_of_states; i++) {
            Arrays.fill(table[i], 0);
        }
        env.reset();
        memory.reset();
    }

    private void planning() {
        for (int i = 0; i < num_of_planning_iterations; i++) {
            int[] sample = env.sample();
            int state = sample[0];
            int action = sample[1];
            int state_next = env.getNextState(state, action);
            float reward = env.getReward(state, action);
            updateTable(state, state_next, action, reward, state_next == num_of_states);
        }

    }

    private void updateTable(int state, int state_next, int action, float reward, boolean last_episode) {
        float expected_return = last_episode ? 0.0f : gamma * table[state_next][getRandomMaxPolicy(state_next)];

        table[state][action] += alpha * (reward + expected_return - table[state][action]);
    }

    private int encodeState(float[] state) {
        if (state == null) {
            return num_of_states;
        } else {
            int state_value = 0;
            for (int i = 0; i < state.length; i++) {
                state_value *= state_resolution;
                double score;
                if (state_ranges[i][0] == Double.NEGATIVE_INFINITY || state_ranges[i][1] == Double.POSITIVE_INFINITY) {
                    score = (state[i] - Math.atan(state_ranges[i][0]))
                            / (Math.atan(state_ranges[i][1]) - Math.atan(state_ranges[i][0]));
                } else {
                    score = (state[i] - state_ranges[i][0]) / (state_ranges[i][1] - state_ranges[i][0]);
                }
                int value = (int) (state_resolution * score);
                if (value >= state_resolution) {
                    value = state_resolution - 1;
                }
                if (value < 0) {
                    value = 0;
                }
                state_value += value;
            }

            return state_value;
        }
    }

    private int getRandomMaxPolicy(int state) {
        int action = -1;
        double action_value = Double.NEGATIVE_INFINITY;
        shuffleIndex();

        for (int i = 0; i < num_of_actions; i++) {
            int proposal = index[i];
            if (action_value < table[state][proposal]) {
                action_value = table[state][proposal];
                action = proposal;
            }
        }
        if (action < 0) {
            throw new IllegalArgumentException("Invalid action");
        }

        return action;
    }

    private void shuffleIndex() {
        for (int i = index.length - 1; i > 0; i--) {
            int k = random.nextInt(i + 1);
            int tmp = index[k];
            index[k] = index[i];
            index[i] = tmp;
        }

    }
}

class EmpiricalModel {
    private final Random random;
    private final int num_of_states;

    private final boolean[] visited_state_mark;
    private final int[] visited_states;
    private final boolean[][] visited_state_action_mark;
    private final int[][] visited_state_actions;

    private final int[][] transitions;
    private final float[][] rewards;

    private final int[] num_of_visited_state_action;
    private int num_of_visited_states = 0;

    public EmpiricalModel(int num_of_states, int num_of_actions, Random random) {
        this.num_of_states = num_of_states;
        this.random = random;

        this.transitions = new int[num_of_states][num_of_actions];
        this.rewards = new float[num_of_states][num_of_actions];

        this.visited_state_mark = new boolean[num_of_states];
        this.visited_states = new int[num_of_states];
        this.visited_state_action_mark = new boolean[num_of_states][num_of_actions];
        this.visited_state_actions = new int[num_of_states][num_of_actions];

        this.num_of_visited_state_action = new int[num_of_states];

        reset();
    }

    public void update(int state, int state_next, int action, float reward, boolean last_episode) {
        if (!visited_state_mark[state]) {
            visited_state_mark[state] = true;
            visited_states[num_of_visited_states] = state;
            num_of_visited_states++;
        }

        if (!visited_state_action_mark[state][action]) {
            visited_state_action_mark[state][action] = true;
            visited_state_actions[state][num_of_visited_state_action[state]] = action;
            num_of_visited_state_action[state]++;
        }
        transitions[state][action] = last_episode ? num_of_states : state_next;
        rewards[state][action] = reward;
    }

    public int getNextState(int state, int action) {
        return transitions[state][action];
    }

    public float getReward(int state, int action) {
        return rewards[state][action];
    }

    public int[] sample() {
        int state = visited_states[random.nextInt(num_of_visited_states)];
        int action = visited_state_actions[state][random.nextInt(num_of_visited_state_action[state])];

        return new int[] { state, action };
    }

    public void reset() {
        Arrays.fill(visited_states, -1);
        Arrays.fill(visited_state_mark, false);
        Arrays.fill(num_of_visited_state_action, 0);
        num_of_visited_states = 0;

        for (int i = 0; i < num_of_states; i++) {
            Arrays.fill(visited_state_action_mark[i], false);
            Arrays.fill(visited_state_actions[i], -1);
            Arrays.fill(transitions[i], 0);
            Arrays.fill(rewards[i], 0.0f);
        }

    }

}
