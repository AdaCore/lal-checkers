procedure Test is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   function F(x : out Integer) return Integer is
   begin
      x := 3;
      return 4;
   end F;

   p : Point := (2, 3, 12);
begin
   p.y := F(p.x);
end Ex1;
