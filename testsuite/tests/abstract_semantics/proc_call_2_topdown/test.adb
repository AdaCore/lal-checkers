procedure Ex1 is
   type Point is record
      x : Integer;
      y : Integer;
      z : Integer;
   end record;

   procedure F(x : out Integer) is
   begin
      x := 3;
   end F;

   p : Point := (2, 3, 12);
begin
   F(p.x);
end Ex1;
