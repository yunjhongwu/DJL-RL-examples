package main.env.http;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import main.env.Environment;
import main.utils.datatype.Snapshot;

public class HttpEnvironment extends Environment {
    private final HttpClient client;
    private final URI address;

    private HttpEnvironment(double[][] state_space, int dim_of_state_space, int num_of_actions, HttpClient client,
            URI address) {
        super(state_space, dim_of_state_space, num_of_actions);
        this.client = client;
        this.address = address;
    }

    public static HttpEnvironment make(String address) {
        HttpClient client = HttpClient.newHttpClient();
        URI address_uri = URI.create(address);
        JsonNode data = sendRequest(Map.of("env", "query"), address_uri, client);

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

        return new HttpEnvironment(state_space, dim, data.get("num_of_actions").intValue(), client, address_uri);

    }

    private static JsonNode sendRequest(Map<String, Object> data, URI address, HttpClient client) {
        try {
            HttpRequest request = HttpRequest.newBuilder(address).version(HttpClient.Version.HTTP_1_1)
                    .POST(HttpRequest.BodyPublishers.ofString(new ObjectMapper().writeValueAsString(data))).build();
            String response = client.send(request, HttpResponse.BodyHandlers.ofString()).body();

            return new ObjectMapper().readTree(response);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void render() {
        // No effect
    }

    @Override
    public void seed(long seed) {
        sendRequest(Map.of("seed", seed), address, client);
    }

    @Override
    public Snapshot reset() {
        return toSnapshot(sendRequest(Map.of("env", "reset"), address, client));
    }

    @Override
    public Snapshot step(int action) {
        return toSnapshot(sendRequest(Map.of("action", action), address, client));
    }

    private Snapshot toSnapshot(JsonNode data) {
        ArrayNode state_node = (ArrayNode) data.get("state");
        float[] state = new float[state_node.size()];

        for (int i = 0; i < state.length; i++) {
            state[i] = state_node.get(i).floatValue();
        }

        return new Snapshot(state, data.get("reward").floatValue(), data.get("mask").asBoolean());
    }

}
