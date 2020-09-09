package main.env.mountaincar;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.geom.Line2D;

import main.env.EnvVisualizer;

public class MountainCarVisualizer extends EnvVisualizer {
    private static final long serialVersionUID = -1L;
    private static final int SCREEN_WIDTH = 600;
    private static final int SCREEN_HEIGHT = 400;
    private static final int CAR_HEIGHT = 20;
    private static final int CAR_WIDTH = 40;
    private static final int WHEEL_DIAMETER = 12;
    private static final int CAR_TOTAL_HEIGHT = CAR_HEIGHT + WHEEL_DIAMETER / 2;

    private final double min_position;
    private final double max_position;
    private final double goal_position;
    private double car_position = -0.5;

    public MountainCarVisualizer(double min_position, double max_position, double goal_position, int fps) {
        super("MountainCar", SCREEN_WIDTH, SCREEN_HEIGHT, fps);
        this.min_position = min_position;
        this.max_position = max_position;
        this.goal_position = goal_position;
    }

    @Override
    protected void paint(Graphics2D g2d) {
        Polygon mountain = new Polygon();

        for (int i = 0; i <= 100; i++) {
            double x_loc = (0.01 * i) * (max_position - min_position) + min_position;
            mountain.addPoint(screenX(x_loc), mountainFunc(x_loc));
        }

        g2d.setColor(Color.BLACK);
        g2d.drawPolyline(mountain.xpoints, mountain.ypoints, mountain.npoints);
        g2d.setStroke(new BasicStroke(2));

        int flag_x = screenX(goal_position);
        int flag_y = mountainFunc(goal_position);
        g2d.setColor(Color.GRAY);
        g2d.fillPolygon(new int[] { flag_x, flag_x, flag_x + SCREEN_WIDTH / 20 },
                new int[] { flag_y - SCREEN_HEIGHT / 10, flag_y - SCREEN_HEIGHT / 15, flag_y - SCREEN_HEIGHT / 15 }, 3);
        g2d.setColor(Color.BLACK);
        g2d.draw(new Line2D.Double(flag_x, flag_y, flag_x, flag_y - SCREEN_HEIGHT / 10));

        int car_x = screenX(car_position);
        int car_y = mountainFunc(car_position);

        g2d.translate(car_x, car_y);
        g2d.rotate(Math.atan(-Math.cos(3 * car_position)));

        g2d.setColor(Color.BLACK);
        g2d.fillRect(-CAR_WIDTH / 2, -CAR_TOTAL_HEIGHT, CAR_WIDTH, CAR_HEIGHT);
        g2d.setColor(Color.GRAY);
        g2d.fillOval(-CAR_WIDTH / 4 - WHEEL_DIAMETER / 2, -WHEEL_DIAMETER, WHEEL_DIAMETER, WHEEL_DIAMETER);
        g2d.fillOval(CAR_WIDTH / 4 - WHEEL_DIAMETER / 2, -WHEEL_DIAMETER, WHEEL_DIAMETER, WHEEL_DIAMETER);
    }

    @Override
    protected void updateState(float[] state) {
        car_position = state[0];
    }

    private int screenX(double x_loc) {
        return (int) ((x_loc - min_position) / (max_position - min_position) * SCREEN_WIDTH);
    }

    private int mountainFunc(double x_loc) {
        return (int) (((1 - Math.sin(3 * x_loc)) * 0.45 * 0.8 + 0.15) * SCREEN_HEIGHT);
    }
}
