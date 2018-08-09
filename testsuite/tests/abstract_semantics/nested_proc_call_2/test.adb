procedure Test is
   x : Integer := 0;
   y : Integer := 42;

   procedure F is
   begin
      x := x + 1;
   end F;

   procedure G is
   begin
      F;
      F;
   end G;
begin
   F;
   F;
   G;
end Ex1;
