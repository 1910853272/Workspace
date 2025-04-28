<<<<<<< HEAD
package com.itheima.d5_data_stream;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

/**
 * 目标：使用数据输入流读取特定类型的数据。
 */
public class DataInputStreamTest2 {
    public static void main(String[] args) {
        try (
                DataInputStream dis =
                        new DataInputStream(new FileInputStream("io-app2/src/itheima10out.txt"));
                ){
            int i = dis.readInt();
            System.out.println(i);

            double d = dis.readDouble();
            System.out.println(d);

            boolean b = dis.readBoolean();
            System.out.println(b);

            String rs = dis.readUTF();
            System.out.println(rs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.d5_data_stream;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

/**
 * 目标：使用数据输入流读取特定类型的数据。
 */
public class DataInputStreamTest2 {
    public static void main(String[] args) {
        try (
                DataInputStream dis =
                        new DataInputStream(new FileInputStream("io-app2/src/itheima10out.txt"));
                ){
            int i = dis.readInt();
            System.out.println(i);

            double d = dis.readDouble();
            System.out.println(d);

            boolean b = dis.readBoolean();
            System.out.println(b);

            String rs = dis.readUTF();
            System.out.println(rs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
