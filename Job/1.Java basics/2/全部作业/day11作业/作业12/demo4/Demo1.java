package cn.iitcast.demo4;

import java.util.Random;
import java.util.Scanner;

public class Demo1 {
    public static void main(String[] args) {
        Number num = new Number();
        new Thread(new RandomNumber(num)).start();
        new Thread(new GuessNumber(num)).start();
    }
}

class RandomNumber implements Runnable {
    private final Number num;

    public RandomNumber(Number num) {
        this.num = num;
    }

    @Override
    public void run() {
        Random r = new Random();
        int number = r.nextInt(100) + 1;
        System.out.println("随机给你一个1-100之间的数，猜猜是多少？");
        while (true) {
            synchronized (num) {
                try {
                    if (num.isFlag() == false) {
                        num.wait();
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                if (num.getNum() == number) {
                    System.out.println("恭喜你，猜中了~");
                    num.setMsg("bingo");
                    num.notify();
                    return;
                } else if (num.getNum() > number) {
                    System.out.println("猜大了~");
                } else if (num.getNum() < number) {
                    System.out.println("猜小了~");
                }
                num.setFlag(false);
                num.notify();
            }
        }
    }
}

class GuessNumber implements Runnable {
    private final Number num;

    public GuessNumber(Number num) {
        this.num = num;
    }

    @Override
    public void run() {
        Scanner sc = new Scanner(System.in);
        int count = 1;
        while (true) {
            synchronized (num) {
                try {
                    if (num.isFlag() == true) {
                        num.wait();
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                if (this.num.getMsg().equals("bingo")) {
                    return;
                }
                int num = sc.nextInt();
                this.num.setNum(num);
                System.out.println("我第" + count++ + "次才的这个数是：" + num);

                this.num.setFlag(true);
                this.num.notify();
            }
        }
    }
}

class Number {
    private int num;
    private String msg = "";
    private boolean flag;

    public boolean isFlag() {
        return flag;
    }

    public void setFlag(boolean flag) {
        this.flag = flag;
    }

    public int getNum() {
        return num;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public void setNum(int num) {
        this.num = num;
    }
}

