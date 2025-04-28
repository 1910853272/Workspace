<<<<<<< HEAD
package com.itheima.demo1;

import java.time.LocalDateTime;

public class Order {
    private int id;
    private String name;
    private LocalDateTime time;
    private double price;

    public Order() {
    }

    public Order(int id, String name, LocalDateTime time, double price) {
        this.id = id;
        this.name = name;
        this.time = time;
        this.price = price;
    }

    /**
     * 获取
     * @return id
     */
    public int getId() {
        return id;
    }

    /**
     * 设置
     * @param id
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * 获取
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * 设置
     * @param name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * 获取
     * @return time
     */
    public LocalDateTime getTime() {
        return time;
    }

    /**
     * 设置
     * @param time
     */
    public void setTime(LocalDateTime time) {
        this.time = time;
    }

    /**
     * 获取
     * @return price
     */
    public double getPrice() {
        return price;
    }

    /**
     * 设置
     * @param price
     */
    public void setPrice(double price) {
        this.price = price;
    }

    @Override
    public String toString() {
        return "Order{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", time=" + time +
                ", price=" + price +
                '}';
    }
}
=======
package com.itheima.demo1;

import java.time.LocalDateTime;

public class Order {
    private int id;
    private String name;
    private LocalDateTime time;
    private double price;

    public Order() {
    }

    public Order(int id, String name, LocalDateTime time, double price) {
        this.id = id;
        this.name = name;
        this.time = time;
        this.price = price;
    }

    /**
     * 获取
     * @return id
     */
    public int getId() {
        return id;
    }

    /**
     * 设置
     * @param id
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * 获取
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * 设置
     * @param name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * 获取
     * @return time
     */
    public LocalDateTime getTime() {
        return time;
    }

    /**
     * 设置
     * @param time
     */
    public void setTime(LocalDateTime time) {
        this.time = time;
    }

    /**
     * 获取
     * @return price
     */
    public double getPrice() {
        return price;
    }

    /**
     * 设置
     * @param price
     */
    public void setPrice(double price) {
        this.price = price;
    }

    @Override
    public String toString() {
        return "Order{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", time=" + time +
                ", price=" + price +
                '}';
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
