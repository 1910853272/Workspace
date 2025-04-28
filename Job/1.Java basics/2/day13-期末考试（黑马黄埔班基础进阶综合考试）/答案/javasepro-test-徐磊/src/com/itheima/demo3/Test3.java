<<<<<<< HEAD
package com.itheima.demo3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Test3 {
    public static void main(String[] args) {
        try (
                // 1、使用流读取文件中的全部数据。
                BufferedReader br = new BufferedReader(new FileReader("javasepro-test-徐磊\\src\\系统菜单.txt"));
                PrintStream ps = new PrintStream("javasepro-test-徐磊\\src\\系统菜单2.txt");
                ){

            // 2、按照行读取读菜单，存入集合中去
            List<String> menus = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                menus.add(line);
            }

            // 3、对菜单进行排序。
            Collections.sort(menus); // 可以

            // 4、遍历集合
            for (String menu : menus) {
                String[] menuNumAndName = menu.split("-");
                System.out.println(menuNumAndName[0].length() == 4 ? menuNumAndName[1] :  "\t" + menuNumAndName[1]);
            }

            // 5、写出菜单到新文件中去
            for (String menu : menus) {
                ps.println(menu);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
=======
package com.itheima.demo3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Test3 {
    public static void main(String[] args) {
        try (
                // 1、使用流读取文件中的全部数据。
                BufferedReader br = new BufferedReader(new FileReader("javasepro-test-徐磊\\src\\系统菜单.txt"));
                PrintStream ps = new PrintStream("javasepro-test-徐磊\\src\\系统菜单2.txt");
                ){

            // 2、按照行读取读菜单，存入集合中去
            List<String> menus = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                menus.add(line);
            }

            // 3、对菜单进行排序。
            Collections.sort(menus); // 可以

            // 4、遍历集合
            for (String menu : menus) {
                String[] menuNumAndName = menu.split("-");
                System.out.println(menuNumAndName[0].length() == 4 ? menuNumAndName[1] :  "\t" + menuNumAndName[1]);
            }

            // 5、写出菜单到新文件中去
            for (String menu : menus) {
                ps.println(menu);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
