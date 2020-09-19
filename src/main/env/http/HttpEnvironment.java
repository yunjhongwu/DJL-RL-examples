package main.env.http;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.http.Consts;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import main.env.Environment;
import main.utils.datatype.Snapshot;

/**
 * The class provides a client to send actions via post requests and receive
 * snapshots from an external http server. Expected post requests including
 * {"env": "query"}, {"env": "reset"}, {"seed": seed}, and "{"action": action}.
 * A valid server should response with {"state": state as a list of floats,
 * "rewards": float rewards, "mask": boolean mask}.
 */
public class HttpEnvironment extends Environment {
    private final HttpClient client;
    private final String address;

    private HttpEnvironment(double[][] state_space, int dim_of_state_space, int num_of_actions, HttpClient client,
            String address) {
        super(state_space, dim_of_state_space, num_of_actions);
        this.client = client;
        this.address = address;
    }

    /**
     * Create a http environment to interact with the specified address.
     * 
     * @param address
     * @return http environment
     */
    public static HttpEnvironment make(String address) {
        HttpClient client = HttpClients.createDefault();
        JsonNode data = sendRequest(Map.of("env", "query"), address, client);
        ArrayNode space = (ArrayNode) data.get("state_space");
        int dim = data.get("dim_of_state_space").intValue();

        double[][] state_space = new double[dim][2];

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < 2; j++) {
                JsonNode node = space.get(i).get(j);
                state_space[i][j] = node.isNull() ? (j == 0 ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY)
                        : node.doubleValue();
            }
        }

        return new HttpEnvironment(state_space, dim, data.get("num_of_actions").intValue(), client, address);
    }

    /**
     * Send prepare and send post requests to the address and handle responses.
     * 
     * @param data
     * @param address
     * @param client
     * @return parsed json data
     */
    private static JsonNode sendRequest(Map<String, Object> data, String address, HttpClient client) {
        try {
            HttpPost post = new HttpPost(address);
            UrlEncodedFormEntity entity = new UrlEncodedFormEntity(data.entrySet().stream()
                    .map(entry -> new BasicNameValuePair(entry.getKey(), String.valueOf(entry.getValue())))
                    .collect(Collectors.toList()), Consts.UTF_8);
            post.setEntity(entity);

            HttpResponse response = client.execute(post);
            int status = response.getStatusLine().getStatusCode();

            if (status >= 200 && status < 300) {
                return new ObjectMapper().readTree(EntityUtils.toString(response.getEntity()));
            } else {
                throw new IOException("Unexpected response status: " + status);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * {@link HttpEnvironment} has no default visualizer.
     */
    @Override
    public void render() {
        // No effect
    }

    /**
     * Send a request of form {"seed": seed} to reset the seed.
     * 
     * @param seed
     */
    @Override
    public void seed(long seed) {
        sendRequest(Map.of("seed", seed), address, client);
    }

    /**
     * Send a request of form {"env": "reset"} to reset the environment.
     * 
     * @param snapshot including next state, reward, and mask.
     */
    @Override
    public Snapshot reset() {
        return toSnapshot(sendRequest(Map.of("env", "reset"), address, client));
    }

    /**
     * Send a request of form {"action": action} to take action.
     */
    @Override
    public Snapshot step(int action) {
        return toSnapshot(sendRequest(Map.of("action", action), address, client));
    }

    /**
     * Convert a valid response to {@link Snapshot}.
     * 
     * @param parsed json data
     * @return snapshot
     */
    private Snapshot toSnapshot(JsonNode data) {
        ArrayNode state_node = (ArrayNode) data.get("state");
        float[] state = new float[state_node.size()];

        for (int i = 0; i < state.length; i++) {
            state[i] = state_node.get(i).floatValue();
        }

        return new Snapshot(state, data.get("reward").floatValue(), data.get("mask").asBoolean());
    }

}
