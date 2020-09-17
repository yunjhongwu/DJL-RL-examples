package main.utils;

import ai.djl.ndarray.NDArray;

public class Helper {
    public static NDArray gather(NDArray arr, int[] indexes) {
        boolean[][] mask = new boolean[(int) arr.size(0)][(int) arr.size(1)];
        for (int i = 0; i < indexes.length; i++) {
            mask[i][indexes[i]] = true;
        }

        return arr.get(arr.getManager().create(mask));
    }
}
