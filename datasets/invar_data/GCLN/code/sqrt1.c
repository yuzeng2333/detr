#include <stdio.h>
#include <assert.h>

int mainQ(int n){
     assert(n >= 0);
  
     int a,s,t;
     a=0;
     s=1;
     t=1;

     int ctr = 0;

     // loop 1
     printf("n, a, s, t\n");     
     while(1){
	  //assert(t == 2*a + 1);
	  //assert(s == (a + 1)*(a + 1));
	  //the above 2 should be equiv to t^2 - 4*s + 2*t + 1 == 0
      //assert(a*a <= n);
	  
	  //%%%traces: int a, int n, int t, int s 
	  if(!(s <= n)) break;
	  a=a+1;
	  t=t+2;
	  s=s+t;
       printf("%d, %d, %d, %d\n", n, a, s, t);    
     }

     return a;
     
}


int main(int argc, char **argv){
     mainQ(atoi(argv[1]));
     return 0;
}

