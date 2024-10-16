
#include <stdio.h>
#include <iostream>

using namespace std;

void call() {
    int a;
    printf("Hello\n");
    std::cout << "Enter a number: ";
    std::cin >> a;
    printf("You entered: %d\n", a);
};


int main(){
    call();
};