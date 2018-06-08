procedure Example1 is
   function Divide(X, Y : Integer) return Integer
     with Pre  => Y /= 0,
          Post => (X > 0 and Divide'Result <= X)
                   or else (X <= 0 and Divide'Result >= X);

   A : Integer;
   B, C : Integer := 0;
   X : Boolean;
begin
   if X then
      A := 10;
      B := 3;
   else
      A := 20;
   end if;
   C := Divide(A, B);
   if C > A then
      C := A;
   end if;
end Example1;
