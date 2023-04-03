#include <stdio.h>
#include <assert.h>
#include <stdlib.h>  //required for afloat to work

//http://www.cs.upc.edu/~erodri/webpage/polynomial_invariants/cohendiv.htm

int mainQ(int x, int y){
     //Cohen's integer division
     //returns x % y

     assert(x>0 && y>0);

     int q=0;
     int r=x;
     int a=0;
     int b=0;
     printf("x, y, q, r, a, b\n");
     // loop 1
     while(1) {
      //assert(x==q*y+r);
      //assert(r>=0);
      //assert((x>=1) && (y>=1));
      ////%%%traces: int x, int y, int q, int a, int b, int r
	  if(!(r>=y)) break;
	  a=1;
	  b=y;
    printf("%d, %d, %d, %d, %d, %d\n", x, y, q, r, a, b);    

      // loop 2
	  while (1){
	       //assert(r>=y*a && b==y*a && x==q*y+r && r>=0 && x>=1 && y>=1);
	       //%%%traces: int x, int y, int q, int a, int b, int r
	       if(!(r >= 2*b)) break;

	       a = 2*a;
	       b = 2*b;
        printf("%d, %d, %d, %d, %d, %d\n", x, y, q, r, a, b);
	  }
	  r=r-b;
	  q=q+a;
    printf("%d, %d, %d, %d, %d, %d\n", x, y, q, r, a, b);    
     }
     //assert(r == x % y);
     //assert(q == x / y);
     //assert(x == q*y+r);
     return q;
}

int main(int argc, char **argv){
     mainQ(atoi(argv[1]), atoi(argv[2]));
     return 0;
}
