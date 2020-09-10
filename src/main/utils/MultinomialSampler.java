package main.utils;

import java.util.Random;

import ai.djl.ndarray.NDArray;

public class MultinomialSampler {
    private static Random RANDOM = new Random();

    public static int exploreExploit(NDArray distribution, Random random, float exploration) {
        if (random.nextFloat() < exploration) {
            return random.nextInt((int) distribution.size());

        } else {
            return (int) distribution.argMax().getLong();
        }
    }

    public static int sample(NDArray distribution, Random random) {
        int value = 0;
        long size = distribution.size();
        float rnd = random.nextFloat();
        for (int i = 0; i < size; i++) {
            float cut = distribution.getFloat(value);
            if (rnd > cut) {
                value++;
            } else {
                return value;
            }
            rnd -= cut;
        }

        throw new IllegalArgumentException("Invalid multinomial distribution");
    }

    public static int sample(NDArray distribution) {
        return sample(distribution, RANDOM);
    }
}
