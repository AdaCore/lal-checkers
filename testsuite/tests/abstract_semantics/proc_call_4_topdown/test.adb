procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x : Integer) return Integer is
   begin
      if x < 42 then
         x := 10;
      else
         x := 20;
      end if;

      if x = 15 then
         return 123;
      end if;

      return x;
   end F;

   p : Point;
begin
   p.y := F(p.x);
end Ex1;
