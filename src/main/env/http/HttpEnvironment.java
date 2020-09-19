package main.env.http;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;
import java.util.Properties;
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

public class HttpEnvironment extends Environment {
    private final HttpClient client;
    private final String address;

    private HttpEnvironment(double[][] state_space, int dim_of_state_space, int num_of_actions, HttpClient client,
            String address) {
        super(state_space, dim_of_state_space, num_of_actions);
        this.client = client;
        this.address = address;
        final Properties properties = System.getProperties();
        properties.setProperty("jdk.internal.httpclient.disableHostnameVerification", Boolean.TRUE.toString());

        properties.setProperty("jdk.httpclient.keepalive.timeout", "0");
    }

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
