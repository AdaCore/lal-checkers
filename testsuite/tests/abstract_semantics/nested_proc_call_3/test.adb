procedure Test is
   procedure F;

   x : Integer := 0;
   y : Integer := 42;

   procedure G is
   begin
      F;
      y := 12;
      F;
   end G;

   procedure F is
   begin
      x := x + 1;
   end F;
begin
   F;
   F;
   G;
end Ex1;
