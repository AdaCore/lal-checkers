procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   procedure G(x : out Integer) is
   begin
      x := 3;
   end G;

   function F(x : out Integer) return Integer is
   begin
      G(x);
      return x + 1;
   end F;

   p : Point;
begin
   p.y := F(p.x);
end Ex1;
